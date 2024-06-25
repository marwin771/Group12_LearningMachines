local sim = require("sim")

string.startswith = function(self, str)
    return self:find("^" .. str) ~= nil
end

local collected_count
local already_collected

function sysCall_init()
    -- do some initialization here:

    -- Make sure you read the section on "Accessing general-type objects programmatically"
    -- For instance, if you wish to retrieve the handle of a scene object, use following instruction:
    -- https://manual.coppeliarobotics.com/en/accessingSceneObjects.htm

    print("Scene initialized")
    collected_count = 0
    already_collected = {}
end

function sysCall_actuation()
    -- put your actuation code here
    --
    -- For example:
    --
    -- local position=sim.getObjectPosition(handle,-1)
    -- position[1]=position[1]+0.001
    -- sim.setObjectPosition(handle,-1,position)
end

function sysCall_sensing()
    -- put your sensing code here
end

function sysCall_cleanup()
    -- do some clean-up here
end

function collect_object(handle)
    -- This may be called multiple times before the object is actually moved
    -- up. The bag checks to eat it, if it's not already eaten. It's not a rabbit
    if not already_collected[handle] then
        print(handle)
        already_collected[handle] = true
        collected_count = collected_count + 1
        print("Collected food " .. collected_count)
    end
end

function sysCall_contactCallback(inData)
    h1 = sim.getObjectName(inData.handle1)
    h2 = sim.getObjectName(inData.handle2)

    if h1:startswith("Food") then 
        collect_food(inData.handle1)
    end

    if h1:startswith("Food") or h1::startswith("Base") then
        print( h1 .. " -> " .. h2)
        if h2:startswith("Food") or h2::startswith("Base") then
            if not already_collected[inData.handle1] or not already_collected[inData.handle2] then
                already_collected[inData.handle1] = true
                already_collected[inData.handle2] = true
                collect_food = collect_food + 1
                print("Food delivired")
            end
        end
    end
    --if h2:startswith("Food") then
    --    print( h1 .. " <- " .. h2)
    --end
    return outData
end

function remote_get_collected_food(inInts, inFloats, inStrings, inBuffer)
    -- inInts, inFloats and inStrings are tables
    -- inBuffer is a string

    -- Perform any type of operation here.

    -- Always return 3 tables and a string, e.g.:
    return { collected_count }, {}, {}, ""
end

-- You can define additional system calls here:
--[[
function sysCall_suspend()
end

function sysCall_resume()
end

function sysCall_dynCallback(inData)
end

function sysCall_jointCallback(inData)
    return outData
end

function sysCall_contactCallback(inData)
    return outData
end

function sysCall_beforeCopy(inData)
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." will be copied")
    end
end

function sysCall_afterCopy(inData)
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." was copied")
    end
end

function sysCall_beforeDelete(inData)
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." will be deleted")
    end
    -- inData.allObjects indicates if all objects in the scene will be deleted
end

function sysCall_afterDelete(inData)
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." was deleted")
    end
    -- inData.allObjects indicates if all objects in the scene were deleted
end
--]]